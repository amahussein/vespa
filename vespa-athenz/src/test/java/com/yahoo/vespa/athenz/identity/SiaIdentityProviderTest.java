// Copyright 2018 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.athenz.identity;

import com.yahoo.security.KeyAlgorithm;
import com.yahoo.security.KeyStoreBuilder;
import com.yahoo.security.KeyStoreType;
import com.yahoo.security.KeyStoreUtils;
import com.yahoo.security.KeyUtils;
import com.yahoo.security.SignatureAlgorithm;
import com.yahoo.security.X509CertificateBuilder;
import com.yahoo.security.X509CertificateUtils;
import com.yahoo.vespa.athenz.api.AthenzService;
import com.yahoo.yolean.Exceptions;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import javax.security.auth.x500.X500Principal;
import java.io.File;
import java.io.IOException;
import java.math.BigInteger;
import java.nio.file.Files;
import java.security.KeyPair;
import java.security.KeyStore;
import java.security.cert.X509Certificate;
import java.time.Duration;
import java.time.Instant;

import static org.junit.Assert.assertNotNull;

/**
 * @author bjorncs
 */
public class SiaIdentityProviderTest {

    @Rule
    public TemporaryFolder tempDirectory = new TemporaryFolder();

    @Test
    public void constructs_ssl_context_from_file() throws IOException {
        File keyFile = tempDirectory.newFile();
        KeyPair keypair = KeyUtils.generateKeypair(KeyAlgorithm.RSA);
        createPrivateKeyFile(keyFile, keypair);

        X509Certificate certificate = createCertificate(keypair);
        File certificateFile = tempDirectory.newFile();
        createCertificateFile(certificate, certificateFile);

        File trustStoreFile = tempDirectory.newFile();
        createTrustStoreFile(certificate, trustStoreFile);

        SiaIdentityProvider provider =
                new SiaIdentityProvider(
                        new AthenzService("domain", "service-name"),
                        keyFile.toPath(),
                        certificateFile.toPath(),
                        trustStoreFile.toPath(),
                        SiaProviderConfig.TrustStoreType.Enum.jks);

        assertNotNull(provider.getIdentitySslContext());
    }

    @Test
    public void constructs_ssl_context_with_pem_trust_store() throws IOException {
        File keyFile = tempDirectory.newFile();
        KeyPair keypair = KeyUtils.generateKeypair(KeyAlgorithm.RSA);
        createPrivateKeyFile(keyFile, keypair);

        X509Certificate certificate = createCertificate(keypair);
        File certificateFile = tempDirectory.newFile();
        createCertificateFile(certificate, certificateFile);

        File trustStoreFile = tempDirectory.newFile();
        createPemTrustStoreFile(certificate, trustStoreFile);

        SiaIdentityProvider provider =
                new SiaIdentityProvider(
                        new AthenzService("domain", "service-name"),
                        keyFile.toPath(),
                        certificateFile.toPath(),
                        trustStoreFile.toPath(),
                        SiaProviderConfig.TrustStoreType.Enum.pem);

        assertNotNull(provider.getIdentitySslContext());
    }

    private void createPrivateKeyFile(File keyFile, KeyPair keypair) throws IOException {
        String privateKeyPem = KeyUtils.toPem(keypair.getPrivate());
        Files.write(keyFile.toPath(), privateKeyPem.getBytes());
    }

    private void createCertificateFile(X509Certificate certificate, File certificateFile) throws IOException {
        String certificatePem = X509CertificateUtils.toPem(certificate);
        Files.write(certificateFile.toPath(), certificatePem.getBytes());
    }

    private X509Certificate createCertificate(KeyPair keypair) {
        Instant now = Instant.now();
        return X509CertificateBuilder
                .fromKeypair(
                        keypair,
                        new X500Principal("CN=subject"),
                        now,
                        now.plus(Duration.ofDays(1)),
                        SignatureAlgorithm.SHA256_WITH_RSA,
                        BigInteger.ONE)
                .build();
    }

    private void createPemTrustStoreFile(X509Certificate certificate, File trustStoreFile) {
        var pemEncoded = X509CertificateUtils.toPem(certificate);
        Exceptions.uncheck(() -> Files.writeString(trustStoreFile.toPath(), pemEncoded));
    }

    private void createTrustStoreFile(X509Certificate certificate, File trustStoreFile) {
        KeyStore keystore = KeyStoreBuilder.withType(KeyStoreType.JKS)
                .withCertificateEntry("dummy-cert", certificate)
                .build();
        KeyStoreUtils.writeKeyStoreToFile(keystore, trustStoreFile.toPath());
    }

}
